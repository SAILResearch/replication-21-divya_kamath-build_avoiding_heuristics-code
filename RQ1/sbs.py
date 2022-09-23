import csv
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:  
        median = (data[size // 2] + data[size // 2 - 1]) / 2
        data[0] = median
    if size % 2 == 1:  
        median = data[(size - 1) // 2]
        data[0] = median
    return data[0]

def output_values(Y_data):
    Y_t = []
    for e in Y_data:
        if e == 'passed':
            Y_t.append(1)
        else:
            Y_t.append(0) 
    return Y_t

def get_data(project_path):
	columns = ['git_num_all_built_commits', 'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_modified', 'tr_status']
	df = pd.read_csv(project_path, usecols = columns)
	src_churn = df['git_diff_src_churn'].tolist()
	file_churn = df['gh_diff_files_modified'].tolist()
	test_churn = df['git_diff_test_churn'].tolist()
	num_commits = df['git_num_all_built_commits'].tolist()
	build_result = output_values(df['tr_status'])

	argument = []
	for index in range(len(src_churn)):
		argument.append([src_churn[index], file_churn[index], test_churn[index], num_commits[index]])

	X = np.array(argument)
	y = np.array(build_result)

	return X, y

def get_duration_data(project_path):
	columns = ['tr_build_id', 'tr_duration']
	df = pd.read_csv(project_path, usecols = columns)
	return df



def with_cv_val():
	#project_names=['rails/rails', 'myronmarston/vcr', 'concerto/concerto', 'benhoskings/babushka', 'rubinius/rubinius', 'rubychan/coderay', 'codeforamerica/adopt-a-hydrant', 'radiant/radiant', 'saberma/shopqi', 'rspec/rspec-core', 'engineyard/engineyard', 'plataformatec/devise', 'rspec/rspec-rails', 'karmi/retire', 'sferik/rails_admin', 'tdiary/tdiary-core', 'dkubb/veritas', 'sstephenson/sprockets', 'thoughtbot/factory_girl', 'weppos/whois', 'errbit/errbit', 'padrino/padrino-framework', 'thoughtbot/paperclip', 'plataformatec/simple_form', 'huerlisi/bookyt', 'hotsh/rstat.us', 'mperham/dalli', 'innoq/iqvoc', 'cheezy/page-object', 'justinfrench/formtastic', 'nov/fb_graph', 'assaf/vanity', 'activerecord-hackery/ransack', 'jimweirich/rake', 'rspec/rspec-mocks', 'neo4jrb/neo4j', 'diaspora/diaspora', 'test-unit/test-unit', 'Shopify/liquid', 'activeadmin/activeadmin', 'ari/jobsworth', 'thoughtbot/shoulda-matchers', 'rubygems/rubygems', 'rdoc/rdoc', 'spree/spree', 'rubyzip/rubyzip', 'pry/pry', 'jruby/activerecord-jdbc-adapter', 'sass/sass', 'jruby/warbler', 'fatfreecrm/fat_free_crm', 'rspec/rspec-expectations', 'excon/excon', 'typus/typus', 'heroku/heroku', 'nahi/httpclient', 'podio/podio-rb', 'maxdemarzi/neography', 'locomotivecms/engine', 'gedankenstuecke/snpr', 'peter-murach/github', 'jnicklas/capybara', 'travis-ci/travis-core', 'presidentbeef/brakeman', 'mikel/mail', 'randym/axlsx', 'kmuto/review', 'danielweinmann/catarse', 'middleman/middleman', 'rubyworks/facets', 'railsbp/rails_best_practices', 'comfy/comfortable-mexican-sofa', 'mongoid/moped', 'wr0ngway/rubber', 'rslifka/elasticity', 'lsegal/yard', 'NoamB/sorcery', 'puppetlabs/puppet', 'mitchellh/vagrant', 'ai/r18n', 'celluloid/celluloid', 'jordansissel/fpm', 'neo4jrb/neo4j-core', 'orbeon/orbeon-forms', 'redis/redis-rb', 'pivotal/pivotal_workstation', 'jruby/jruby', 'louismullie/treat', 'puma/puma', 'pophealth/popHealth', 'twitter/twitter-cldr-rb', 'gistflow/gistflow', 'adamfisk/LittleProxy', 'awestruct/awestruct', 'jnunemaker/httparty', 'Graylog2/graylog2-server', 'neuland/jade4j', 'sensu/sensu', 'shawn42/gamebox', 'applicationsonline/librarian', 'haml/haml', 'sporkmonger/addressable', 'google/google-api-ruby-client', 'elm-city-craftworks/practicing-ruby-web', 'sunlightlabs/scout', 'floere/phony', 'data-axle/cassandra_object', 'typhoeus/typhoeus', 'shoes/shoes4', 'troessner/reek', 'recurly/recurly-client-ruby', 'CloudifySource/cloudify', 'puppetlabs/puppetlabs-firewall', 'typhoeus/ethon', 'sparklemotion/nokogiri', 'tinkerpop/blueprints', 'tinkerpop/rexster', 'thinkaurelius/titan', 'openSUSE/open-build-service', 'engineyard/ey-cloud-recipes', 'git/git-scm.com', 'honeybadger-io/honeybadger-ruby', 'azagniotov/stubby4j', 'sferik/twitter', 'calagator/calagator', 'openshift/rhc', 'codefirst/AsakusaSatellite', 'DatabaseCleaner/database_cleaner', 'burke/zeus', 'fog/fog', 'twilio/twilio-java', 'twitter/commons', 'Albacore/albacore', 'prawnpdf/prawn', 'enspiral/loomio', 'refinery/refinerycms', 'sevntu-checkstyle/sevntu.checkstyle', 'opal/opal', 'graphhopper/graphhopper', 'sparklemotion/mechanize', 'SomMeri/less4j', 'tent/tentd', 'searchbox-io/Jest', 'square/dagger', 'google/truth', 'square/okhttp', 'square/retrofit', 'maxcom/lorsource', 'jneen/rouge', 'jmkgreen/morphia', 'SpontaneousCMS/spontaneous', 'everzet/capifony', 'killbill/killbill', 'scobal/seyren', 'intuit/simple_deploy', 'projectblacklight/blacklight', 'rapid7/metasploit-framework', 'amahi/platform', 'vcr/vcr', 'Findwise/Hydra', 'structr/structr', 'sachin-handiekar/jInstagram', 'nutzam/nutz', 'slim-template/slim', 'puppetlabs/puppetlabs-stdlib', 'puppetlabs/facter', 'phoet/on_ruby', 'dreamhead/moco', 'travis-ci/travis.rb', 'cloudfoundry/cloud_controller_ng', 'square/assertj-android', 'jmxtrans/jmxtrans', 'twitter/secureheaders', 'nanoc/nanoc', 'expertiza/expertiza', 'asciidoctor/asciidoctor', 'rubber/rubber', 'openMF/mifosx', 'mybatis/mybatis-3', 'test-kitchen/test-kitchen', 'owlcs/owlapi', 'engineyard/engineyard-serverside', 'selendroid/selendroid', 'ruboto/ruboto', 'openfoodfoundation/openfoodnetwork', 'stephanenicolas/robospice', 'joscha/play-authenticate', 'undera/jmeter-plugins', 'cantino/huginn', 'resque/resque', 'albertlatacz/java-repl', 'l0rdn1kk0n/wicket-bootstrap', 'dynjs/dynjs', 'abarisain/dmix', 'dropwizard/dropwizard', 'dropwizard/metrics', 'jberkel/sms-backup-plus', 'rubymotion/sugarcube', 'naver/yobi', 'Shopify/active_shipping', 'projecthydra/sufia', 'rubymotion/BubbleWrap', 'pivotal-sprout/sprout-osx-apps', 'chef/omnibus', 'JodaOrg/joda-time', 'EmmanuelOga/ffaker', 'kostya/eye', 'laurentpetit/ccw', 'puniverse/quasar', 'simpligility/android-maven-plugin', 'jsonld-java/jsonld-java', 'travis-ci/travis-cookbooks', 'FenixEdu/fenixedu-academic', 'threerings/playn', 'restlet/restlet-framework-java', 'jedi4ever/veewee', 'sensu/sensu-community-plugins', 'OpenRefine/OpenRefine', 'chef/chef', 'fluent/fluentd', 'perwendel/spark', 'joelittlejohn/jsonschema2pojo', 'jOOQ/jOOQ', 'springside/springside4', 'github/hub', 'johncarl81/parceler', 'discourse/onebox', 'julianhyde/optiq', 'ruby-ldap/ruby-net-ldap', 'DSpace/DSpace', 'jeremyevans/sequel', 'bikeindex/bike_index', 'doanduyhai/Achilles', 'rackerlabs/blueflood', 'rodjek/librarian-puppet', 'p6spy/p6spy', 'square/wire', 'Nodeclipse/nodeclipse-1', 'rebelidealist/stripe-ruby-mock', 'checkstyle/checkstyle', 'elastic/logstash', 'airlift/airlift', 'lenskit/lenskit', 'MiniProfiler/rack-mini-profiler', 'geoserver/geoserver', 'ocpsoft/rewrite', 'Unidata/thredds', 'torakiki/pdfsam', 'loopj/android-async-http', 'feedbin/feedbin', 'recruit-tech/redpen', 'brettwooldridge/HikariCP', 'puppetlabs/marionette-collective', 'iipc/openwayback', 'caelum/vraptor4', 'dianping/cat', 'jphp-compiler/jphp', 'mockito/mockito', 'oblac/jodd', 'facebook/buck', 'facebook/presto', 'jpos/jPOS', 'hamstergem/hamster', 'mongodb/morphia', 'realestate-com-au/pact', 'inaturalist/inaturalist', 'jtwig/jtwig', 'go-lang-plugin-org/go-lang-idea-plugin', 'square/picasso', 'voltrb/volt', 'zxing/zxing', 'openaustralia/morph', 'GlowstoneMC/Glowstone', 'owncloud/android', 'JakeWharton/u2020', 'rpush/rpush', 'OneBusAway/onebusaway-android', 'rabbit-shocker/rabbit', 'azkaban/azkaban', 'relayrides/pushy', 'deeplearning4j/deeplearning4j', 'github/developer.github.com', 'xetorthio/jedis', 'FasterXML/jackson-core', 'FasterXML/jackson-databind', 'protostuff/protostuff', 'atmos/heaven', 'MrTJP/ProjectRed', 'lemire/RoaringBitmap', 'apache/drill', 'Kapeli/cheatsheets', 'gradle/gradle', 'OpenGrok/OpenGrok', 'spring-io/sagan', 'mendhak/gpslogger', 'thoughtbot/hound', 'teamed/qulice', 'jcabi/jcabi-aspects', 'jcabi/jcabi-github', 'jcabi/jcabi-http', 'yegor256/rultor', 'querydsl/querydsl', 'codevise/pageflow', 'grails/grails-core', 'weld/core', 'thatJavaNerd/JRAW', 'bndtools/bnd', 'igniterealtime/Openfire', 'zendesk/samson', 'bndtools/bndtools', 'xtreemfs/xtreemfs', 'puniverse/capsule', 'broadinstitute/picard', 'github/github-services', 'gavinlaking/vedeu', 'haiwen/seadroid', 'AChep/AcDisplay', 'GoClipse/goclipse', 'hsz/idea-gitignore', 'jsprit/jsprit', 'dblock/waffle', 'numenta/htm.java', 'rightscale/praxis', 'google/error-prone', 'datastax/ruby-driver', 'iluwatar/java-design-patterns', 'Netflix/Hystrix', 'oyachai/HearthSim', 'jayway/JsonPath', 'exteso/alf.io', 'spring-cloud/spring-cloud-config', 'validator/validator', 'HubSpot/jinjava', 'connectbot/connectbot', 'google/physical-web', 'myui/hivemall', 'MarkUsProject/Markus', 'jMonkeyEngine/jmonkeyengine', 'davidmoten/rxjava-jdbc', 'qos-ch/logback', 'Homebrew/homebrew-science', 'GoogleCloudPlatform/DataflowJavaSDK', 'SoftInstigate/restheart', 'naver/pinpoint', 'KronicDeth/intellij-elixir', 'embulk/embulk', 'loomio/loomio', 'openstreetmap/openstreetmap-website', 'activescaffold/active_scaffold', 'tananaev/traccar', 'SonarSource/sonarqube', 'grpc/grpc-java', 'psi-probe/psi-probe', 'orientation/orientation', 'square/keywhiz', 'aws/aws-sdk-java', 'Shopify/shipit-engine', 'perfectsense/brightspot-cms', 'jamesagnew/hapi-fhir']

	project_names=['rails.csv', 'jruby.csv', 'metasploit-framework.csv', 'heroku.csv', 'vagrant.csv', 'opal.csv', 'cloudify.csv', 'cloud_controller_ng.csv', 'rubinius.csv', 'open-build-service.csv', 'gradle.csv', 'sonarqube.csv', 'loomio.csv', 'fog.csv', 'puppet.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv', 'graylog2-server.csv']

	for project in project_names[:]:
		
		string = "../data/even_data/first_failures/train/" + project.split('.')[0] + "_train.csv"
		X, y = get_data(string)

		KF = KFold(n_splits=10)


		less = 0
		more = 0
		yes = 0

		#print(X)
		num_test = 0
		num_feature=4

		precision = []
		recall = []
		f1 = []
		build_save = []
		fitted_model = []


		for train_index, test_index in KF.split(X):
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = y[train_index], y[test_index]

			num_test = num_test + len(Y_test)
			X_train = X_train.reshape((int(len(X_train)), num_feature))

			try:
				rf = RandomForestClassifier(class_weight={0:0.05,1:1})
				predictor = rf.fit(X_train, Y_train)
			except:
				rf = RandomForestClassifier()
				predictor = rf.fit(X_train, Y_train)

			X_test = X_test.reshape((int(len(X_test)), num_feature))
			Y_result=(predictor.predict(X_test))


			'''for index in range(len(Y_result)):
				if Y_result[index]==0 and Y_test[index]==0 and Y_test[index-1]==1 and Y_result[index-1]==1:
					yes=yes+1
				if Y_result[index] == 0 and Y_result[index-1]==1:
					more=more+1
				if Y_test[index] == 0 and Y_test[index-1]==1:
					less=less+1

			if less != 0:
				recall0 = yes/less
				if more == 0:
					precision0=1
				else:
					precision0 = yes / more'''

			precision0 = precision_score(Y_test, Y_result)
			recall0 = recall_score(Y_test, Y_result)
			f10 = f1_score(Y_test, Y_result)

			precision.append(precision0)
			recall.append(recall0)
			f1.append(f10)
			fitted_model.append(rf)


		best_f1 = max(f1)
		max_index = f1.index(best_f1)

		print(f1)
		print(best_f1)

		best_fit_model = fitted_model[max_index]

		string = "../data/even_data/first_failures/test/" + project.split('.')[0] + "_test.csv"
		X_val, y_val = get_data(string)

		y_pred = best_fit_model.predict(X_val)
		print(y_pred)
		print(y_val)

		print(precision_score(y_val, y_pred))
		print(recall_score(y_val, y_pred))

		#Since we have already divided train and test data, we don't need to collect build ids again
		result_df = get_duration_data(string)
		result_df['Build_Result'] = y_pred
		result_df['Actual_Result'] = y_val
		result_df['Index'] = list(range(1, len(y_val)+1))

		#print(commit_values)
		headers = ['tr_build_ids', 'tr_duration','Duration', 'Build_Result', 'Actual_Result']

		file_name = './' + project.split('.')[0] + '_abcd_metrics.csv'
		result_df.to_csv(file_name)


def without_cv_val():
	#project_names=['rails/rails', 'myronmarston/vcr', 'concerto/concerto', 'benhoskings/babushka', 'rubinius/rubinius', 'rubychan/coderay', 'codeforamerica/adopt-a-hydrant', 'radiant/radiant', 'saberma/shopqi', 'rspec/rspec-core', 'engineyard/engineyard', 'plataformatec/devise', 'rspec/rspec-rails', 'karmi/retire', 'sferik/rails_admin', 'tdiary/tdiary-core', 'dkubb/veritas', 'sstephenson/sprockets', 'thoughtbot/factory_girl', 'weppos/whois', 'errbit/errbit', 'padrino/padrino-framework', 'thoughtbot/paperclip', 'plataformatec/simple_form', 'huerlisi/bookyt', 'hotsh/rstat.us', 'mperham/dalli', 'innoq/iqvoc', 'cheezy/page-object', 'justinfrench/formtastic', 'nov/fb_graph', 'assaf/vanity', 'activerecord-hackery/ransack', 'jimweirich/rake', 'rspec/rspec-mocks', 'neo4jrb/neo4j', 'diaspora/diaspora', 'test-unit/test-unit', 'Shopify/liquid', 'activeadmin/activeadmin', 'ari/jobsworth', 'thoughtbot/shoulda-matchers', 'rubygems/rubygems', 'rdoc/rdoc', 'spree/spree', 'rubyzip/rubyzip', 'pry/pry', 'jruby/activerecord-jdbc-adapter', 'sass/sass', 'jruby/warbler', 'fatfreecrm/fat_free_crm', 'rspec/rspec-expectations', 'excon/excon', 'typus/typus', 'heroku/heroku', 'nahi/httpclient', 'podio/podio-rb', 'maxdemarzi/neography', 'locomotivecms/engine', 'gedankenstuecke/snpr', 'peter-murach/github', 'jnicklas/capybara', 'travis-ci/travis-core', 'presidentbeef/brakeman', 'mikel/mail', 'randym/axlsx', 'kmuto/review', 'danielweinmann/catarse', 'middleman/middleman', 'rubyworks/facets', 'railsbp/rails_best_practices', 'comfy/comfortable-mexican-sofa', 'mongoid/moped', 'wr0ngway/rubber', 'rslifka/elasticity', 'lsegal/yard', 'NoamB/sorcery', 'puppetlabs/puppet', 'mitchellh/vagrant', 'ai/r18n', 'celluloid/celluloid', 'jordansissel/fpm', 'neo4jrb/neo4j-core', 'orbeon/orbeon-forms', 'redis/redis-rb', 'pivotal/pivotal_workstation', 'jruby/jruby', 'louismullie/treat', 'puma/puma', 'pophealth/popHealth', 'twitter/twitter-cldr-rb', 'gistflow/gistflow', 'adamfisk/LittleProxy', 'awestruct/awestruct', 'jnunemaker/httparty', 'Graylog2/graylog2-server', 'neuland/jade4j', 'sensu/sensu', 'shawn42/gamebox', 'applicationsonline/librarian', 'haml/haml', 'sporkmonger/addressable', 'google/google-api-ruby-client', 'elm-city-craftworks/practicing-ruby-web', 'sunlightlabs/scout', 'floere/phony', 'data-axle/cassandra_object', 'typhoeus/typhoeus', 'shoes/shoes4', 'troessner/reek', 'recurly/recurly-client-ruby', 'CloudifySource/cloudify', 'puppetlabs/puppetlabs-firewall', 'typhoeus/ethon', 'sparklemotion/nokogiri', 'tinkerpop/blueprints', 'tinkerpop/rexster', 'thinkaurelius/titan', 'openSUSE/open-build-service', 'engineyard/ey-cloud-recipes', 'git/git-scm.com', 'honeybadger-io/honeybadger-ruby', 'azagniotov/stubby4j', 'sferik/twitter', 'calagator/calagator', 'openshift/rhc', 'codefirst/AsakusaSatellite', 'DatabaseCleaner/database_cleaner', 'burke/zeus', 'fog/fog', 'twilio/twilio-java', 'twitter/commons', 'Albacore/albacore', 'prawnpdf/prawn', 'enspiral/loomio', 'refinery/refinerycms', 'sevntu-checkstyle/sevntu.checkstyle', 'opal/opal', 'graphhopper/graphhopper', 'sparklemotion/mechanize', 'SomMeri/less4j', 'tent/tentd', 'searchbox-io/Jest', 'square/dagger', 'google/truth', 'square/okhttp', 'square/retrofit', 'maxcom/lorsource', 'jneen/rouge', 'jmkgreen/morphia', 'SpontaneousCMS/spontaneous', 'everzet/capifony', 'killbill/killbill', 'scobal/seyren', 'intuit/simple_deploy', 'projectblacklight/blacklight', 'rapid7/metasploit-framework', 'amahi/platform', 'vcr/vcr', 'Findwise/Hydra', 'structr/structr', 'sachin-handiekar/jInstagram', 'nutzam/nutz', 'slim-template/slim', 'puppetlabs/puppetlabs-stdlib', 'puppetlabs/facter', 'phoet/on_ruby', 'dreamhead/moco', 'travis-ci/travis.rb', 'cloudfoundry/cloud_controller_ng', 'square/assertj-android', 'jmxtrans/jmxtrans', 'twitter/secureheaders', 'nanoc/nanoc', 'expertiza/expertiza', 'asciidoctor/asciidoctor', 'rubber/rubber', 'openMF/mifosx', 'mybatis/mybatis-3', 'test-kitchen/test-kitchen', 'owlcs/owlapi', 'engineyard/engineyard-serverside', 'selendroid/selendroid', 'ruboto/ruboto', 'openfoodfoundation/openfoodnetwork', 'stephanenicolas/robospice', 'joscha/play-authenticate', 'undera/jmeter-plugins', 'cantino/huginn', 'resque/resque', 'albertlatacz/java-repl', 'l0rdn1kk0n/wicket-bootstrap', 'dynjs/dynjs', 'abarisain/dmix', 'dropwizard/dropwizard', 'dropwizard/metrics', 'jberkel/sms-backup-plus', 'rubymotion/sugarcube', 'naver/yobi', 'Shopify/active_shipping', 'projecthydra/sufia', 'rubymotion/BubbleWrap', 'pivotal-sprout/sprout-osx-apps', 'chef/omnibus', 'JodaOrg/joda-time', 'EmmanuelOga/ffaker', 'kostya/eye', 'laurentpetit/ccw', 'puniverse/quasar', 'simpligility/android-maven-plugin', 'jsonld-java/jsonld-java', 'travis-ci/travis-cookbooks', 'FenixEdu/fenixedu-academic', 'threerings/playn', 'restlet/restlet-framework-java', 'jedi4ever/veewee', 'sensu/sensu-community-plugins', 'OpenRefine/OpenRefine', 'chef/chef', 'fluent/fluentd', 'perwendel/spark', 'joelittlejohn/jsonschema2pojo', 'jOOQ/jOOQ', 'springside/springside4', 'github/hub', 'johncarl81/parceler', 'discourse/onebox', 'julianhyde/optiq', 'ruby-ldap/ruby-net-ldap', 'DSpace/DSpace', 'jeremyevans/sequel', 'bikeindex/bike_index', 'doanduyhai/Achilles', 'rackerlabs/blueflood', 'rodjek/librarian-puppet', 'p6spy/p6spy', 'square/wire', 'Nodeclipse/nodeclipse-1', 'rebelidealist/stripe-ruby-mock', 'checkstyle/checkstyle', 'elastic/logstash', 'airlift/airlift', 'lenskit/lenskit', 'MiniProfiler/rack-mini-profiler', 'geoserver/geoserver', 'ocpsoft/rewrite', 'Unidata/thredds', 'torakiki/pdfsam', 'loopj/android-async-http', 'feedbin/feedbin', 'recruit-tech/redpen', 'brettwooldridge/HikariCP', 'puppetlabs/marionette-collective', 'iipc/openwayback', 'caelum/vraptor4', 'dianping/cat', 'jphp-compiler/jphp', 'mockito/mockito', 'oblac/jodd', 'facebook/buck', 'facebook/presto', 'jpos/jPOS', 'hamstergem/hamster', 'mongodb/morphia', 'realestate-com-au/pact', 'inaturalist/inaturalist', 'jtwig/jtwig', 'go-lang-plugin-org/go-lang-idea-plugin', 'square/picasso', 'voltrb/volt', 'zxing/zxing', 'openaustralia/morph', 'GlowstoneMC/Glowstone', 'owncloud/android', 'JakeWharton/u2020', 'rpush/rpush', 'OneBusAway/onebusaway-android', 'rabbit-shocker/rabbit', 'azkaban/azkaban', 'relayrides/pushy', 'deeplearning4j/deeplearning4j', 'github/developer.github.com', 'xetorthio/jedis', 'FasterXML/jackson-core', 'FasterXML/jackson-databind', 'protostuff/protostuff', 'atmos/heaven', 'MrTJP/ProjectRed', 'lemire/RoaringBitmap', 'apache/drill', 'Kapeli/cheatsheets', 'gradle/gradle', 'OpenGrok/OpenGrok', 'spring-io/sagan', 'mendhak/gpslogger', 'thoughtbot/hound', 'teamed/qulice', 'jcabi/jcabi-aspects', 'jcabi/jcabi-github', 'jcabi/jcabi-http', 'yegor256/rultor', 'querydsl/querydsl', 'codevise/pageflow', 'grails/grails-core', 'weld/core', 'thatJavaNerd/JRAW', 'bndtools/bnd', 'igniterealtime/Openfire', 'zendesk/samson', 'bndtools/bndtools', 'xtreemfs/xtreemfs', 'puniverse/capsule', 'broadinstitute/picard', 'github/github-services', 'gavinlaking/vedeu', 'haiwen/seadroid', 'AChep/AcDisplay', 'GoClipse/goclipse', 'hsz/idea-gitignore', 'jsprit/jsprit', 'dblock/waffle', 'numenta/htm.java', 'rightscale/praxis', 'google/error-prone', 'datastax/ruby-driver', 'iluwatar/java-design-patterns', 'Netflix/Hystrix', 'oyachai/HearthSim', 'jayway/JsonPath', 'exteso/alf.io', 'spring-cloud/spring-cloud-config', 'validator/validator', 'HubSpot/jinjava', 'connectbot/connectbot', 'google/physical-web', 'myui/hivemall', 'MarkUsProject/Markus', 'jMonkeyEngine/jmonkeyengine', 'davidmoten/rxjava-jdbc', 'qos-ch/logback', 'Homebrew/homebrew-science', 'GoogleCloudPlatform/DataflowJavaSDK', 'SoftInstigate/restheart', 'naver/pinpoint', 'KronicDeth/intellij-elixir', 'embulk/embulk', 'loomio/loomio', 'openstreetmap/openstreetmap-website', 'activescaffold/active_scaffold', 'tananaev/traccar', 'SonarSource/sonarqube', 'grpc/grpc-java', 'psi-probe/psi-probe', 'orientation/orientation', 'square/keywhiz', 'aws/aws-sdk-java', 'Shopify/shipit-engine', 'perfectsense/brightspot-cms', 'jamesagnew/hapi-fhir']

	project_names=['rails.csv', 'jruby.csv', 'metasploit-framework.csv', 'heroku.csv', 'vagrant.csv', 'opal.csv', 'cloudify.csv', 'cloud_controller_ng.csv', 'rubinius.csv', 'open-build-service.csv', 'gradle.csv', 'sonarqube.csv', 'loomio.csv', 'fog.csv', 'puppet.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv', 'graylog2-server.csv']
	
	for project in project_names[:]:
		
		string = "../data/datasets/all_datasets/" + project
		X, y = get_data(string)

		KF = KFold(n_splits=10)


		less = 0
		more = 0
		yes = 0

		#print(X)
		num_test = 0
		num_feature=4

		precision = []
		recall = []
		build_save = []
		fitted_model = []

		for train_index, test_index in KF.split(X):
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = y[train_index], y[test_index]

			num_test = num_test + len(Y_test)
			X_train = X_train.reshape((int(len(X_train)), num_feature))

			try:
				rf = RandomForestClassifier(class_weight={0:0.05,1:1})
				predictor = rf.fit(X_train, Y_train)
			except:
				rf = RandomForestClassifier()
				predictor = rf.fit(X_train, Y_train)

			X_test = X_test.reshape((int(len(X_test)), num_feature))
			Y_result=(predictor.predict(X_test))


			for index in range(len(Y_result)):
				if Y_result[index]==0 and Y_test[index]==0 and Y_test[index-1]==1 and Y_result[index-1]==1:
					yes=yes+1
				if Y_result[index] == 0 and Y_result[index-1]==1:
					more=more+1
				if Y_test[index] == 0 and Y_test[index-1]==1:
					less=less+1

			if less != 0:
				recall0 = yes/less
				if more == 0:
					precision0=1
				else:
					precision0 = yes / more

			precision0 = precision_score(Y_test, Y_result)
			recall0 = recall_score(Y_test, Y_result)

			precision.append(precision0)
			recall.append(recall0)
			build_save.append(1-more/num_test)
			fitted_model.append(rf)


		best_precision = max(precision)
		max_index = precision.index(best_precision)

		print(precision)
		print(best_precision)

		best_fit_model = fitted_model[max_index]

		string = "../data/even_data/test/" + project.split('.')[0] + "_test.csv"
		X_val, y_val = get_data(string)

		y_pred = best_fit_model.predict(X_val)
		print(y_pred)
		print(y_val)

		print(precision_score(y_val, y_pred))
		print(recall_score(y_val, y_pred))

		#Since we have already divided train and test data, we don't need to collect build ids again
		result_df = get_duration_data(string)
		result_df['Build_Result'] = y_pred
		result_df['Actual_Result'] = y_val
		result_df['Index'] = list(range(1, len(y_val)+1))

		#print(commit_values)
		headers = ['tr_build_ids', 'tr_duration','Duration', 'Build_Result', 'Actual_Result']

		file_name = './' + project.split('.')[0] + '_abcd_metrics.csv'
		result_df.to_csv(file_name)


#with_cv_val()






















def validation():
	project_names=['rails.csv', 'jruby.csv', 'metasploit-framework.csv', 'heroku.csv', 'vagrant.csv', 'opal.csv', 'cloudify.csv', 'cloud_controller_ng.csv', 'rubinius.csv', 'open-build-service.csv', 'gradle.csv', 'sonarqube.csv', 'loomio.csv', 'fog.csv', 'puppet.csv', 'concerto.csv', 'sufia.csv', 'geoserver.csv', 'orbeon-forms.csv', 'graylog2-server.csv']
	
	batchsize = [1,2,4,8,16,32]
	batch_result = 'results/batch_results/final_result.csv'
	final_file = open(batch_result, 'w')
	final_headers = ['project', 'method', 'batch', 'reqd_builds', 'delay']
	final_writer = csv.writer(final_file)
	final_writer.writerow(final_headers)

	for project in project_names:

		for b in batchsize:
		

			#file to write results of SBS algorithm
			proj_result = 'results/batch_results/batch_' + str(b) + '_' + project.split('.')[0] + '_result.csv'
			result_file = open(proj_result, 'w')
			res_headers = ['index', 'duration', 'total_builds']
			res_writer = csv.writer(result_file)
			res_writer.writerow(res_headers)

			file_name = './' + project.split('.')[0] + '_abcd_metrics.csv'

			csv_file = pd.read_csv(file_name)

			actual_results = csv_file['Actual_Result'].tolist()
			pred_results = csv_file['Build_Result'].tolist()

			delay_indexes = []
			built_indexes = []
			first_failure = 0
			ci = []
			
			total_builds = len(actual_results)
			sbs_builds = 0

			for i in range(len(actual_results)):

				#If first failure is already found, continue building until actual build pass is seen
				if first_failure == 1:
					ci.append(0)
					sbs_builds += 1

					if actual_results[i] == 1:
						#actual build pass is seen, switch to prediction
						first_failure = 0
					else:
						first_failure = 1
				else:
					#we're in prediction state, if predicted to skip, we skip
					if pred_results[i] == 1:
						ci.append(1)
					else:
						#if predicted to fail, we switch to determine state and set first_failure to True
						ci.append(0)
						sbs_builds += 1
						first_failure = 1-actual_results[i]


			total_builds = len(ci)
			actual_builds = ci.count(0)

			saved_builds = 100*ci.count(1)/total_builds
			reqd_builds = 100*ci.count(0)/total_builds
			
			for i in range(len(ci)):
				if ci[i] == 0:
					built_indexes.append(i)
				else:
					if actual_results[i] == 0:
						delay_indexes.append(i)

			
			from_value = 0
			delay = []
			for k in range(len(built_indexes)):
				for j in range(len(delay_indexes)):
					if delay_indexes[j] > from_value and delay_indexes[j] < built_indexes[k]:
						delay.append(built_indexes[k] - delay_indexes[j])
				from_value = built_indexes[k]

			final_index = len(ci)

			for j in range(len(delay_indexes)):
				if delay_indexes[j] > from_value and delay_indexes[j] < final_index:
					delay.append(final_index - delay_indexes[j])



			print('saved_builds for {} is {}'.format(project, saved_builds))
			print('delay for {} is {}\n\n'.format(project, sum(delay)))
			final_writer.writerow([project, 'sbs', b, reqd_builds, sum(delay)])


			durations = csv_file['tr_duration'].tolist()
			batch_size = b
			batch_builds = 0
			commit_num = 1
			build_time = 0

			for i in range(len(ci)):

				if commit_num == batch_size:
					res_writer.writerow([i+1, build_time, batch_builds])
					commit_num = 1
					build_time = 0
					batch_builds = 0
					continue

				if ci[i] == 0:
					batch_builds += 1
					build_time += durations[i]

				commit_num += 1

		 
			# file_name = 'metrics/' + project.split('.')[0] + '_real_metrics.csv'

			# csv_file = csv.reader(open(file_name, 'r'))

			# built_commits = []
			# build_time = 0
			# total_builds = 0
			# actual_builds = 0
			# commit_num = 1
			# flag = 0
			# batches = []
			# num = 0
			# b_id = 0
			




			# 	# if a build is predicted to fail, they will build it
			# 	if build[-2] == '0':
			# 		#add the build time
			# 		build_time += int(build[2])
			# 		actual_builds += 1
			# 		total_builds += 1
			# 		b_id = build[0]
			# 		flag = 1

			# 	#if prev build has failed, build until you see a true build pass
			# 	if flag == 1:
			# 		if build[-1] == '0':
			# 			if b_id != build[0]:
			# 				build_time += int(build[2])
			# 				actual_builds += 1
			# 				total_builds += 1				
			# 		if build[-1] == '1':
			# 			#this is the first build pass after failure
			# 			#go back to predicting
			# 			if b_id != build[0]:
			# 				build_time += int(build[2])
			# 				actual_builds += 1
			# 				total_builds += 1
			# 			flag = 0


			# 	'''#if a build passes,
			# 	if build[-2] == '1':
			# 		#check if this is the first build pass after failure
			# 		if (flag == 1):
			# 			flag = 0
			# 			build_time += int(build[2])
			# 			total_builds += 1'''

			# 	if commit_num == 4:
			# 		batches.append([int(build[1]), build_time, total_builds])
			# 		res_writer.writerow([int(build[1]), build_time, total_builds])
			# 		commit_num = 0
			# 		built_commits.append(build_time)
			# 		build_time = 0
			# 		total_builds = 0

			# 	commit_num += 1
			# #print(batches)
			# #print(total_builds)
			# print(actual_builds)
			# #print(len(csv_file))
			# #print('Total time taken for builds:')
			# #print(built_commits)

validation()